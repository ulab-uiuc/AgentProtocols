# AgentConnect: https://github.com/agent-network-protocol/AgentConnect
# Author: GaoWei Chang
# Email: chgaowei@gmail.com
# Website: https://agent-network-protocol.com/
#
# This project is open-sourced under the MIT License. For details, please see the LICENSE file.

import json
import uuid
import asyncio
import unittest
from typing import Optional, Awaitable, Callable
from datetime import datetime, timedelta

from tests.test_code.generated_code.generated_requester_code import UserEducationRequester

class TestUserEducationRequester(unittest.TestCase):
    """Test cases for UserEducationRequester class"""

    def setUp(self):
        """Set up test fixtures"""
        self.requester = UserEducationRequester()
        self.sent_message = None
        self.requester.set_send_callback(self._mock_send_callback)

    async def _mock_send_callback(self, message: bytes) -> None:
        """Mock callback to capture sent message and generate response"""
        self.sent_message = message
        # 解析发送的消息
        request = json.loads(message.decode('utf-8'))
        
        # 根据请求构造相应的响应
        if request['userId'] == 'invalid_id':
            response = self._create_error_response(request['messageId'], 400, "Invalid user_id format")
        elif request['userId'] == 'not_found':
            response = self._create_error_response(request['messageId'], 404, "User not found")
        else:
            response = self._create_success_response(request)
            
        # 模拟异步响应
        await asyncio.sleep(0.1)
        self.requester.handle_message(json.dumps(response).encode('utf-8'))

    def _create_success_response(self, request: dict) -> dict:
        """Create a success response with mock education data"""
        today = datetime.now()
        education_data = [
            {
                "institution": "Test University",
                "major": "Computer Science",
                "degree": "Bachelor",
                "achievements": "Outstanding Graduate",
                "startDate": (today - timedelta(days=1460)).strftime("%Y-%m-%d"),
                "endDate": (today - timedelta(days=365)).strftime("%Y-%m-%d")
            }
        ]
        
        if request.get('includeDetails', False):
            education_data[0]["achievements"] = "Dean's List, Research Award, Academic Excellence"
            
        return {
            "messageType": "getUserEducation",
            "messageId": request['messageId'],
            "code": 200,
            "data": education_data,
            "pagination": {
                "currentPage": request.get('page', 1),
                "totalPages": 1,
                "totalItems": 1
            }
        }

    def _create_error_response(self, message_id: str, code: int, error_message: str) -> dict:
        """Create an error response"""
        return {
            "messageType": "getUserEducation",
            "messageId": message_id,
            "code": code,
            "error": {
                "message": error_message
            }
        }

    def _verify_request_format(self, request_str: str) -> None:
        """验证请求格式是否符合协议规范"""
        request = json.loads(request_str)
        
        # 验证必需字段
        self.assertIn('messageType', request)
        self.assertEqual(request['messageType'], 'getUserEducation')
        self.assertIn('messageId', request)
        self.assertIn('userId', request)
        
        # 验证可选字段类型
        if 'includeDetails' in request:
            self.assertIsInstance(request['includeDetails'], bool)
        if 'page' in request:
            self.assertIsInstance(request['page'], int)
            self.assertGreaterEqual(request['page'], 1)
        if 'pageSize' in request:
            self.assertIsInstance(request['pageSize'], int)
            self.assertGreaterEqual(request['pageSize'], 1)

    async def test_successful_request(self):
        """Test successful education data retrieval"""
        success, data = await self.requester.send_request(
            user_id=str(uuid.uuid4()),
            include_details=True,
            page=1,
            page_size=10
        )
        
        # 验证请求格式
        self._verify_request_format(self.sent_message.decode('utf-8'))
        
        # 验证响应
        self.assertTrue(success)
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        
        # 验证教育数据字段
        education = data[0]
        required_fields = ['institution', 'major', 'degree', 'startDate', 'endDate']
        for field in required_fields:
            self.assertIn(field, education)
            
        # 验证详细信息
        self.assertIn('achievements', education)
        self.assertTrue(len(education['achievements']) > 20)  # 详细信息应该更长

    async def test_invalid_user_id(self):
        """Test request with invalid user ID"""
        success, data = await self.requester.send_request(
            user_id='invalid_id'
        )
        
        print(f"success: {success}, data: {data}")

        self.assertFalse(success)
        # self.assertIn('error', data)
        self.assertEqual(data['message'], 'Invalid user_id format')

    async def test_user_not_found(self):
        """Test request for non-existent user"""
        success, data = await self.requester.send_request(
            user_id='not_found'
        )
        
        self.assertFalse(success)
        # self.assertIn('error', data)
        self.assertEqual(data['message'], 'User not found')

    async def test_pagination(self):
        """Test pagination parameters"""
        success, data = await self.requester.send_request(
            user_id=str(uuid.uuid4()),
            page=2,
            page_size=5
        )
        
        # 验证请求中的分页参数
        request = json.loads(self.sent_message.decode('utf-8'))
        self.assertEqual(request['page'], 2)
        self.assertEqual(request['pageSize'], 5)

    async def test_invalid_parameters(self):
        """Test invalid parameter validation"""
        with self.assertRaises(ValueError):
            await self.requester.send_request(user_id='', page=0)

def run_tests():
    """Run all test cases"""
    async def run_async_tests():
        test_cases = [
            TestUserEducationRequester('test_successful_request'),
            TestUserEducationRequester('test_invalid_user_id'),
            TestUserEducationRequester('test_user_not_found'),
            TestUserEducationRequester('test_pagination')
            # TestUserEducationRequester('test_invalid_parameters')
        ]
        
        for test_case in test_cases:
            test_case.setUp()
            test_method = getattr(test_case, test_case._testMethodName)
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
    
    asyncio.run(run_async_tests())

if __name__ == '__main__':
    run_tests() 